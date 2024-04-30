docker build --platform linux/amd64 -t docker-image:test .
docker tag docker-image:test XXXX.dkr.ecr.us-west-2.amazonaws.com/YYYY:latest
docker push XXXX.dkr.ecr.us-west-2.amazonaws.com/YYYY:latest