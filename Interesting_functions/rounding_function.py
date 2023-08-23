#!/usr/bin/env python3
def rounding(x):
	result=x-int(x)
	if(x-int(x))>=0.5:
		return int(x) + 1
	else:
		return int(x)

