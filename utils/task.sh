#!/bin/bash # employ bash shell

daily_excute="TRUE" # define execute period
hourly_excute="FALSE"

if ["$daily_excute" = "TRUE"]; then
  for file in 'ls /usr/Revery-Recommendation/utils'
  do
  #skip the shell script if it's empty
  if [-f $file] ; then
    if['ls -l $file|awk "{print $5}"' -gt 0] ; then
      python main.py --online true --visualize false & >> trending.log
    fi
  fi
  done
fi

echo "excution started in daily $daily_excute | or hourly $hourly_excute" # echo is used to printf in terminal

