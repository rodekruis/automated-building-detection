# automated-building-detection
Automated building detection tools

### Requirements:
1. 

## Getting started
### Using Docker
1. Install [Docker](https://www.docker.com/get-started).
2. Download the [latest Docker Image](https://hub.docker.com/r/jmargutti/automated-building-detection)
```
docker pull jmargutti/automated-building-detection
```
3. Create a docker container
```
docker run --name automated-building-detection -dit -v <path-to-your-workspace>:/workspace -p 5000:5000 jmargutti/automated-building-detection
```
4. Access the container
```
docker exec -it automated-building-detection bash
```

### Manual Setup
TBI

