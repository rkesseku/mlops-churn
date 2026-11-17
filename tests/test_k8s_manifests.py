"""Structural checks for the Kubernetes manifests (no cluster or kubectl needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

K8S_DIR = Path(__file__).resolve().parents[1] / "k8s"
YAML_FILES = sorted(K8S_DIR.rglob("*.yaml"))


def load_all(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def base_docs() -> list[dict]:
    """Every manifest listed in base/kustomization.yaml, plus the optional MLflow component."""
    kust = yaml.safe_load((K8S_DIR / "base" / "kustomization.yaml").read_text())
    docs: list[dict] = []
    for name in kust["resources"]:
        docs += load_all(K8S_DIR / "base" / name)
    return docs


def find(docs: list[dict], kind: str, name: str) -> dict:
    for d in docs:
        if d["kind"] == kind and d["metadata"]["name"] == name:
            return d
    raise AssertionError(f"{kind}/{name} not found")


def test_manifests_exist():
    assert YAML_FILES, "no k8s manifests found"


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: str(p.relative_to(K8S_DIR)))
def test_every_yaml_parses(path):
    docs = load_all(path)
    assert docs
    for d in docs:
        assert "apiVersion" in d and "kind" in d


def test_kustomization_resources_exist():
    for kfile in K8S_DIR.rglob("kustomization.yaml"):
        k = yaml.safe_load(kfile.read_text())
        for ref in k.get("resources", []) + k.get("components", []):
            assert (kfile.parent / ref).exists(), f"{kfile}: missing {ref}"


def test_secret_template_not_deployed_and_has_only_placeholders():
    kust = yaml.safe_load((K8S_DIR / "base" / "kustomization.yaml").read_text())
    assert "secret.template.yaml" not in kust["resources"]
    secret = load_all(K8S_DIR / "base" / "secret.template.yaml")[0]
    assert all(v == "REPLACE_ME" for v in secret["stringData"].values())


def test_serving_deployment_fields():
    dep = find(base_docs(), "Deployment", "churn-serving")
    tmpl = dep["spec"]["template"]
    # selector must match pod labels
    sel = dep["spec"]["selector"]["matchLabels"]
    assert sel.items() <= tmpl["metadata"]["labels"].items()
    c = tmpl["spec"]["containers"][0]
    assert c["image"].startswith("churn-serving:")
    for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
        assert c[probe]["httpGet"]["path"] == "/health"
    assert c["ports"][0]["containerPort"] == 8000
    for section in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(c["resources"][section])
    ann = tmpl["metadata"]["annotations"]
    assert ann["prometheus.io/scrape"] == "true"
    assert ann["prometheus.io/port"] == "8000"
    assert ann["prometheus.io/path"] == "/metrics"
    assert tmpl["spec"]["securityContext"]["runAsNonRoot"] is True


def test_service_selects_serving_pods_and_port_is_named():
    docs = base_docs()
    svc = find(docs, "Service", "churn-serving")
    dep = find(docs, "Deployment", "churn-serving")
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert svc["spec"]["selector"].items() <= labels.items()
    port_names = {p["name"] for p in dep["spec"]["template"]["spec"]["containers"][0]["ports"]}
    assert svc["spec"]["ports"][0]["targetPort"] in port_names


def test_hpa_targets_serving_deployment():
    docs = base_docs()
    hpa = find(docs, "HorizontalPodAutoscaler", "churn-serving")
    ref = hpa["spec"]["scaleTargetRef"]
    assert (ref["kind"], ref["name"]) == ("Deployment", "churn-serving")
    find(docs, "Deployment", ref["name"])
    assert hpa["spec"]["minReplicas"] <= hpa["spec"]["maxReplicas"]


def test_cronjob_runs_retrain_module():
    docs = base_docs()
    cj = find(docs, "CronJob", "churn-retrain")
    assert cj["spec"]["concurrencyPolicy"] == "Forbid"
    assert len(cj["spec"]["schedule"].split()) == 5
    pod = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    assert pod["restartPolicy"] in ("Never", "OnFailure")
    c = pod["containers"][0]
    assert c["image"].startswith("churn-retrain:")
    assert c["command"] == ["python", "-m", "churn_platform.retrain"]
    # volume mounts reference declared volumes backed by an existing PVC
    vols = {v["name"]: v for v in pod["volumes"]}
    for m in c["volumeMounts"]:
        assert m["name"] in vols
    pvc_names = {d["metadata"]["name"] for d in docs if d["kind"] == "PersistentVolumeClaim"}
    assert vols["data"]["persistentVolumeClaim"]["claimName"] in pvc_names


def test_configmap_and_envfrom_references_resolve():
    docs = base_docs()
    cm = find(docs, "ConfigMap", "churn-config")
    assert "MLFLOW_TRACKING_URI" in cm["data"]
    assert all(isinstance(v, str) for v in cm["data"].values())  # ConfigMap values must be strings
    for d in docs:
        if d["kind"] in ("Deployment", "CronJob"):
            tmpl = (
                d["spec"]["template"]
                if d["kind"] == "Deployment"
                else d["spec"]["jobTemplate"]["spec"]["template"]
            )
            for c in tmpl["spec"]["containers"]:
                for ef in c.get("envFrom", []):
                    if "configMapRef" in ef:
                        assert ef["configMapRef"]["name"] == "churn-config"
                    if "secretRef" in ef:
                        assert ef["secretRef"]["name"] == "churn-secrets"


def test_every_namespaced_resource_uses_the_namespace():
    for d in base_docs():
        if d["kind"] != "Namespace":
            assert d["metadata"]["namespace"] == "mlops-churn", d["metadata"]["name"]


def test_pushgateway_service_matches_deployment():
    docs = base_docs()
    dep = find(docs, "Deployment", "pushgateway")
    svc = find(docs, "Service", "pushgateway")
    assert svc["spec"]["selector"].items() <= dep["spec"]["template"]["metadata"]["labels"].items()


def test_mlflow_component_service_matches_deployment():
    docs = load_all(K8S_DIR / "components" / "mlflow" / "mlflow.yaml")
    dep = find(docs, "Deployment", "mlflow")
    svc = find(docs, "Service", "mlflow")
    assert svc["spec"]["selector"].items() <= dep["spec"]["template"]["metadata"]["labels"].items()
    # service name must match the default tracking URI host in the ConfigMap
    cm = find(base_docs(), "ConfigMap", "churn-config")
    assert cm["data"]["MLFLOW_TRACKING_URI"].startswith("http://mlflow:")


def test_local_overlay_patch_targets_exist():
    k = yaml.safe_load((K8S_DIR / "overlays" / "local" / "kustomization.yaml").read_text())
    docs = base_docs()
    for p in k["patches"]:
        find(docs, p["target"]["kind"], p["target"]["name"])
        assert yaml.safe_load(p["patch"])  # patch body is valid YAML
