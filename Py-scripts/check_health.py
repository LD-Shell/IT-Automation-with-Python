#!/usr/bin/env python3
#This script gives the user information about the computer's health
import shutil #File and dir access module
import psutil #CPU usage
import pandas #Module for dataframe setup


#Codeblock to analyze disk usage

du=shutil.disk_usage('/')
du_total=du.total
du_used=du.used
du_free=du.free
free_space=du.free/du.total*100
storage=[du_total,du_used,du_free,free_space]

#Codeblock to analyze memory usage
cpu_percent=psutil.cpu_percent(0.1)


