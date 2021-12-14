# Vision-campus
Docker based ML / CV project skeleton
## Create Development Environment in PyCharm
Clone new project and create run/debug configurations with below settings in PyCharm
### Add Docker Server
Open Preferences > Build,Execution,Deployment > Docker : Add docker server
### Create Dockerfile Configuration and Build Image / Create Container
Create new run/debug configuration from Dockerfile template with below settings:
- Image Tag: vision-campus
- Container name: vision-campus
- Context folder: .
- Bind ports: 8888:8888
- Bind mounts: {project-root-path}:/opt/project
- Run build image: checked  

Run this file to build image and create container
### Create Docker Remote Interpreter
**_Image building should be finished for create docker remote interpreter._**  
Open Preferences > Project > Python interpreter > Add Python Interpreter : Select "Docker" and set "Image name" as "vision-campus" then click "Ok"  
Configure "Path mappings" setting in Python Interpreter: Open "Edit Project Path Mappings" dialog window  
Add new Path Mappings with below settings:
- Local path: {project-root-path}
- Remote path: /opt/project
### Create Flask Server Configuration and Start (apps/service.py)
Create new run/debug configuration from "Flask server" template with below settings:
- Target type: Script path
- Target: {project-root-path}/apps/service.py
- FLASK_ENV: development
- FLASK_DEBUG: checked
- Python Interpreter: vision-campus:latest
- Docker container settings 
    - Port bindings: 5000:5000
    - Volume bindings: {project-root-path}:/opt/project
  
Run this file to start flask server on container
### Before Run Python App,Download yolo_campus files
First of all,you have to download this files on google drive.
-train-datasett/obj.names
-yolov3.cfg
-backup/yolov3_final.weights
-IMG_4333.JPG
You have to move this files to weights/yolo_campus/
### Run Python App on Running Container in PyCharm
Right click on running container at "Services" window and then select option to "Create terminal" or "Exec"  
Write that command template for run python app in exec or terminal: "python {path-of-app-file}"
- Run client.py: python apps/client.py

## Create Development Environment in Terminal
Clone new project and create new image and container in project root path
### Build Image
```
docker build -t vision-campus .
```
### Create Container
```
docker run -v {project-root-path}:/opt/project -p 8888:8888 --name vision-campus -it vision-campus
```
### Access Jupyter Notebook Server Token and Links
```
docker exec vision-campus jupyter notebook list
```
### Run Python App in Container
```
docker exec {container-name} python {path-of-app-file}
```
Run client.py app on Flask (apps/service.py) container:
```
docker exec {flask-service-container-id} python apps/client.py
```
Run main.py app on vision-campus container:
```
docker exec vision-campus python apps/main.py
```