#!/usr/bin/env bash

TENSORBOARDS=$(ps aux | grep '[t]ensorboard' | awk '{print $2}')
echo $TENSORBOARDS "will be killed, hit [Enter] to confirm"
read
kill $TENSORBOARDS