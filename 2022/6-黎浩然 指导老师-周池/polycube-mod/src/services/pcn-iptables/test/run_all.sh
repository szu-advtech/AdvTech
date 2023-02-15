#!/bin/bash

echo "" > log.txt
echo "" > results.txt

for f in $(find . -name '*.sh')
    do
    if [[ $f != *"run_all.sh"* ]];then
        echo "executing test $f ..."

        sudo polycubed &>/dev/null  2>&1 &
        sleep 10

        bash "$f" -H >> log.txt
        result=$(echo $?)
        echo "test $f -> $result"
        echo "test $f -> $result" >> results.txt
        sudo pkill -SIGTERM polycubed
        sleep 10
    fi
done
