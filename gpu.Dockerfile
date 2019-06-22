# the gpu image: docker run --runtime=nvidia -it --net=host --ipc=host -p 3006:3006 -v <host_dir>:/ds hamelsmu/ml-gpu-lite
# this image (cpu): https://cloud.docker.com/u/github/repository/docker/github/mdtok
FROM hamelsmu/ml-gpu-lite

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY notebooks notebooks/

EXPOSE 7654
CMD ["sh", "-c", "jupyter notebook --no-browser --allow-root --port=7654 --NotebookApp.token='$pass'"]