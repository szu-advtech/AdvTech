#! /bin/bash

# 1 pcn-bridge
# 1 loop link on the bridge
# 2 interfaces veth1 to  forward traffic
# test connectivity, after convergence

source "${BASH_SOURCE%/*}/../../helpers.bash"

function cleanup {
  set +e
  del_bridges 1
  delete_veth 2
  delete_link 1
}
trap cleanup EXIT

set -x

# setup
create_veth 2
create_link 1

set -e

add_bridges_stp 1

# create ports
bridge_add_port br1 link11
bridge_add_port br1 link12
bridge_add_port br1 veth1
bridge_add_port br1 veth2

#sleeping
sleep $CONVERGENCE_TIME

# test ports state
test_forwarding_pcn br1 link11
test_blocking_pcn br1 link12

test_forwarding_pcn br1 veth1
test_forwarding_pcn br1 veth2


# change topology
polycubectl br1 ports del link11
polycubectl br1 ports del link12

bridge_add_port br1 link12
bridge_add_port br1 link11

#sleeping
sleep $CONVERGENCE_TIME

# test ports state
test_forwarding_pcn br1 link12
test_blocking_pcn br1 link11

test_forwarding_pcn br1 veth1
test_forwarding_pcn br1 veth2
