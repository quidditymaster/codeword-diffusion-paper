
docker run --rm -it \
-v $(pwd):/data \
-w /data \
-u $(id -u ${USER}):$(id -g ${USER}) \
tianon/latex /bin/bash -c "pdflatex -halt-on-error $1"