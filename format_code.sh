#!/bin/sh

black *.py -l 100
black lensless/*.py -l 100
black scripts/*.py -l 100
black scripts/recon/*.py -l 100
black profile/*.py -l 100
black test/*.py -l 100
black docs/source/*.py -l 100
