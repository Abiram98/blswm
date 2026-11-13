# Stage 1: source of Virtuoso binaries, entrypoint, etc.
FROM openlink/virtuoso-opensource-7:7.2.14-r21-g092d7fb-ubuntu AS virtuoso

# Stage 2: final image with Python + PyTorch + CUDA + Virtuoso
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime AS virt_torch

# (Optional but recommended) – make sure base OS has typical tools
# The PyTorch image is already Ubuntu-based, so this is usually cheap.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        locales \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Copy Virtuoso installation and entrypoint from the first stage
COPY --from=virtuoso /opt/virtuoso-opensource /opt/virtuoso-opensource
COPY --from=virtuoso /virtuoso-entrypoint.sh /virtuoso-entrypoint.sh

# Copy the OpenSSL 1.1 libs that Virtuoso was built against
COPY --from=virtuoso /usr/lib/x86_64-linux-gnu/libssl.so.1.1 /usr/lib/x86_64-linux-gnu/
COPY --from=virtuoso /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 /usr/lib/x86_64-linux-gnu/


#Allegedly viral but not tested https://virtuoso.openlinksw.com/whitepapers/Configuring%20Virtuoso%20for%20Scale.html
#COPY --from=virtuoso /usr/share/virtuoso/vad/ /usr/share/virtuoso/vad/
#COPY --from=virtuoso /usr/lib/virtuoso/hosting/ /usr/lib/virtuoso/hosting/
#COPY --from=virtuoso /var/lib/virtuoso/vsp/ /var/lib/virtuoso/vsp/


# Virtuoso expects a /database volume; ensure it exists
RUN mkdir -p /database && \
    chmod +x /virtuoso-entrypoint.sh

# Environment normally provided by the official Virtuoso image
# (you can override these at `docker run` with `-e` if needed)
ENV DBA_PASSWORD=dba \
    SPARQL_UPDATE=true \
    DEFAULT_GRAPH=http://example.com \
    PATH="/opt/virtuoso-opensource/bin:${PATH}"

COPY virtuoso.ini /database/virtuoso.ini
RUN echo 'export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' >> /root/.bashrc

# Virtuoso ports
EXPOSE 1111 8890
VOLUME ["/database"]

CMD ["/bin/bash"]

FROM virt_torch AS final_ml
RUN apt-get update -y
RUN apt-get -y install maven
RUN apt-get -y install openjdk-17-jdk openjdk-17-jre
RUN apt-get install graphviz graphviz-dev -y
#setup python dependencies
RUN pip install ray[data,train,tune,serve]
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
RUN pip install matplotlib==3.8.2
RUN pip install notebook==7.0.6
RUN pip install rdflib==7.0.0
RUN pip install scikit-learn==1.3.2
RUN pip install SPARQLWrapper==2.0.0
RUN pip install JPype1==1.5.0

FROM final_ml AS dev_ml
# Setup ssh for development
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install collectl
RUN apt-get install ssh -y
RUN apt-get install rsync -y
RUN apt install openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test
RUN  echo 'root:test' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

CMD ["/bin/bash", "-lc", "service ssh start && exec /bin/bash"]

