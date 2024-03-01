import asyncio
import sys
import time
import socket
import struct
import threading
from datetime import datetime

def timestamp(name, stage):
    print(f'[{datetime.now()}] {name}, {stage}, {time.time():.3f}', file=sys.stderr)

class Package:
    head_ip: str # str(len('192.168.249.123')) = 15
    port : int
    word_size : int
    rank : int
    tp : int
    pp : int
    start_id : int
    pipeline_stage_size: int
    wg : float
    wc : float
    cg : float
    cc : float
#     tensor_ranks : List[int]
#     piprline_ranks : List[int]
    def __init__(self, head_ip,port,word_size,rank,tp,pp,start_id,pipeline_stage_size,
                wg,wc,cg,cc):  
        self.head_ip = head_ip
        self.port = port
        self.word_size = word_size
        self.rank = rank
        self.tp = tp
        self.pp = pp
        self.start_id = start_id
        self.pipeline_stage_size = pipeline_stage_size
        self.wg = wg
        self.wc = wc
        self.cg = cg
        self.cc = cc
        
    def pack(self):
        return struct.pack('15s7i4f', bytes(self.head_ip, 'utf-8'),self.port,self.word_size,self.rank,self.tp,self.pp,self.start_id,self.pipeline_stage_size,
                          self.wg,self.wc,self.cg,self.cc)
    @classmethod
    def unpack(cls, data):
        head_ip,port,word_size,rank,tp,pp,start_id,pipeline_stage_size,wg,wc,cg,cc = struct.unpack('15s7i4f', data)
        return cls(head_ip,port,word_size,rank,tp,pp,start_id,pipeline_stage_size,
                wg,wc,cg,cc)
    
class TcpServer():
    def __init__(self, address, port, blocking=True):
        self.address = address
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.address, self.port))
        self.sock.listen(1)
        self.sock.setblocking(blocking)
        self.receive_all = False
        self.send_package = {}

    def __del__(self):
        self.sock.close()

    def accept(self):
        conn, address = self.sock.accept()
        return conn, address

    async def async_accept(self):
        loop = asyncio.get_running_loop()
        conn, address = await loop.sock_accept(self.sock)
        return conn, address


class TcpAgent:
    def __init__(self, conn, address=None, blocking=True):
        self.conn = conn
        self.address = address
        # self.conn.setblocking(blocking)

    def __del__(self):
        self.conn.close()

    def send(self, msg):
        self.conn.sendall(msg)

    def recv(self, msg_len):
        return self.conn.recv(msg_len, socket.MSG_WAITALL)

    async def async_send(self, msg):
        while msg:
            try:
                sent = self.conn.send(msg)
            except socket.error as e:
                if e.errno == socket.errno.EWOULDBLOCK:
                    await asyncio.sleep(0)
                    continue
                raise
            msg = msg[sent:]

    async def async_recv(self, msg_len):
        msg = b''
        while len(msg) < msg_len:
            try:
                chunk = self.conn.recv(msg_len - len(msg))
            except socket.error as e:
                if e.errno == socket.errno.EWOULDBLOCK:
                    await asyncio.sleep(0)
                    continue
                raise
            if not chunk:
                raise Exception('Connection closed by remote end')
            msg += chunk
        return msg

    def send_string(self, s):
        data = s.encode('utf-8')
        l = len(data)
        self.conn.sendall(struct.pack('I', l))
        self.conn.sendall(data)

    def recv_string(self):
        l = self.recv(4)
        l, = struct.unpack('I', l)
        data = self.recv(l)
        return data.decode('utf-8')


    def settimeout(self, t):
        self.conn.settimeout(t)


class TcpClient(TcpAgent):
    def __init__(self, address, port):
        super().__init__(None)
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((address, port))

class ServerThread(threading.Thread):
    def __init__(self, tcp_server, agent_dict) -> None:
        super().__init__()
        self.tcp_server = tcp_server
        self.agent_dict = agent_dict
        self.received_heartbeat_time = {}

    def run(self):
        timestamp('gs TcpThread', 'listening')
        while True:
            conn, address = self.tcp_server.accept()
            agent = TcpAgent(conn, address)
            data = agent.recv(4)
            rank, = struct.unpack('I', data)
            timestamp('gs', f'accepted from pc rank {rank}')
            self.agent_dict[str(rank)] = agent
            self.received_heartbeat_time[str(rank)] = time.time()
            if self.tcp_server.receive_all:
                send_data = self.tcp_server.send_package[rank].pack()
                agent.send(send_data)

class ClientThread(threading.Thread):
    def __init__(self, tcp_server_address,tcp_server_port,rank) -> None:
        super().__init__()
        self.setting = None
        self.rank = rank
        self.tcp_server_address = tcp_server_address
        self.tcp_server_port = tcp_server_port
    def run(self):
        try: 
            # 发送心跳 并实时接收最新的状态
            while True:
                data = struct.pack('I', self.rank)
                tcpClient = TcpClient(self.tcp_server_address,self.tcp_server_port)
                tcpClient.send(data)
                print("send")
                data = tcpClient.recv(1024)
                if len(data) == 0:
                    continue
                self.setting = Package.unpack(data)
                print(setting)
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")


  
def send_heartbeat(args):
    while True:
        data = struct.pack('I', args.rank)
        # 这个占时很久
        tcpClient = TcpClient(args.tcp_server_address,args.tcp_server_port)
        tcpClient.send(data)
        print("send")
        data = tcpClient.recv(1024)
        if len(data) == 0:
            continue
        args.setting = Package.unpack(data)
        print(setting)
        time.sleep(1)


