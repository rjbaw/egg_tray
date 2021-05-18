import socket

PORT = 8080
IP = '127.0.0.1'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((IP, PORT))
s.listen()
print('ADDRESS: tcp://{}:{}'.format(IP, PORT))
connection, address = s.accept()

#connection.setblocking(0)
print(connection.recv(2048))
while True:
    try:
#        print(connection.recv(2048))
        connection.sendall('hello from server'.encode('ascii'))
    except Exception as e:
        print(e)
