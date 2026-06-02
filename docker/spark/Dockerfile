# Custom Spark image:
#   - Apache Spark 3.5.3 official base (PySpark included, Python 3.11)
#   - hadoop-aws + aws-java-sdk JARs for S3A access to MinIO
#   - Our Python source on the PYTHONPATH

FROM apache/spark:3.5.3

USER root

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262

RUN curl -fSLo /opt/spark/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
        https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
 && curl -fSLo /opt/spark/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
        https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar

RUN pip install --no-cache-dir \
    pydantic==2.9.2 \
    pydantic-settings==2.6.1

# Make Spark binaries directly callable: `spark-submit`, `pyspark`, etc.
ENV PATH="/opt/spark/bin:/opt/spark/sbin:${PATH}"

# Add our source to the Python path so jobs can `from churn_platform.config import settings`.
ENV PYTHONPATH=/opt/jobs/src

# Apache image runs as user 'spark' (uid 185); keep that.
USER 185
