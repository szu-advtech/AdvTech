sh ./test_servera.sh & PIDA=$!
sh ./test_serverb.sh & PIDB=$!
wait $PIDA
wait $PIDB