# Egg tray feature detector
## Directory Setup
### WSL
```
sudo apt-get install git wget
git clone https://github.com/ezvk7740/egg_tray.git
cd egg_tray
mkdir weights
cd weights
wget https://www.dropbox.com/s/5bg9yis4kqtouud/egg_tray.pt
cd ..
```
## Conda Environment Setup
```
conda create --name eggtray python=3.7
conda activate eggtray
conda install pytorch==1.4.0 torchvision==0.5.0 scipy matplotlib tqdm numpy cudatoolkit=10.1 -c pytorch
```
## Run
### Normal operation
```
python get_coordinate.py
```
### Fixing Bugs (optional)
add this line here to the file at C:\Users\{your USERNAME}\.conda\envs\eggtray\Lib\sites-packages\torch\array.py
```
def __array__(self, dtype=None):
	if dtype is None:
		return self.cpu().numpy()
	else:
		return self.numpy().as type(dtype, copy=False)
```
## Arguments
### Socket
Disable-socket
```
python get_coordinate.py --disable-socket
```
Testing socket
```
python socket_server.py
python get_coordinate.py --test-socket
```
### Source 
Selecting sources (use integer only)

e.g.
```
python get_coordinate.py --source 0

```
### Changing IP/PORT
Modify this part in get_coordinate.py
```
PORT = 54600
IP = '192.168.1.10'
```
### Weights
Changing weights path

e.g.
```
python get_coordinate.py --weights weights/hello.pt
```
### GPU/CPU
use gpu `python get_coordinate.py --device ''`
use cpu`python get_coordinate.py --device cpu`

