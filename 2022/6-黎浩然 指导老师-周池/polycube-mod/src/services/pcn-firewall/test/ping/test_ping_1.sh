# PING 1 testing corner cases 63rd and 64th rule matched.

source "${BASH_SOURCE%/*}/../helpers.bash"

batch='{"rules":['

function fwsetup {
  polycubectl firewall add fw
  polycubectl attach fw veth1
  polycubectl firewall fw chain INGRESS set default=DROP
  polycubectl firewall fw chain EGRESS set default=DROP
}

function fwcleanup {
  set +e
  polycubectl firewall del fw
  delete_veth 2
}
trap fwcleanup EXIT

echo -e '\nTest 1 \n'
set -e
set -x

TYPE="TC"

if [ -n "$1" ]; then
  TYPE=$1
fi

create_veth 2

fwsetup

set +x
batch='{"rules":['

#INGRESS CHAIN
#dumb rules
for i in `seq 0 62`;
do
  batch=${batch}"{'operation': 'append', 'src': '10.1.$((i%2)).$((i%255))/31', 'dst': '10.1.$((i%2)).$((i%255))', 'l4proto': 'TCP', 'sport': $i, 'dport': $i, 'tcpflags': 'SYN', 'action': 'DROP'},"
done

#matched rules
batch=${batch}"{'operation': 'insert', 'id': 63, 'src': '10.0.0.1', 'dst': '10.0.0.2', 'l4proto': 'ICMP', 'action': 'ACCEPT'},"

#dumb rules
for i in `seq 64 128`;
do
  batch=${batch}"{'operation': 'append', 'src': '11.1.0.$((i%255))', 'dst': '11.1.0.$((i%255))', 'l4proto': 'TCP', 'sport': $i, 'dport': $i, 'action': 'DROP'},"
done

batch=${batch}"]}"
set -x
polycubectl firewall fw chain INGRESS batch rules=<<<$batch
set +x
batch='{"rules":['

#EGRESS CHAIN
#dumb rules
for i in `seq 0 63`;
do
  batch=${batch}"{'operation': 'append', 'src': '10.1.$((i%2)).$((i%255))/32', 'dst': '10.1.$((i%2)).$((i%255))', 'l4proto': 'TCP', 'sport': $i, 'dport': $i, 'tcpflags': 'SYN', 'action': 'DROP'},"
done

#matched rules
batch=${batch}"{'operation': 'insert', 'id': 64, 'src': '10.0.0.2/32', 'dst': '10.0.0.1/32', 'l4proto': 'ICMP', 'action': 'ACCEPT'},"

#dumb rules
for i in `seq 65 129`;
do
  batch=${batch}"{'operation': 'append', 'src': '11.1.0.$((i%255))', 'dst': '11.1.0.$((i%255))/16', 'l4proto': 'TCP', 'sport': $i, 'dport': $i, 'tcpflags': '!ACK', 'action': 'DROP'},"
done

batch=${batch}"]}"
set -x
polycubectl firewall fw chain EGRESS batch rules=<<<$batch

#ping
sudo ip netns exec ns1 ping 10.0.0.2 -c 2 -i 0.5 -w 1
