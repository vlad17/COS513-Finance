#!/bin/bash
#
# Sets everything (cluster and summary) running with some
# hard-coded directory values.

set -x

GCF=/n/fs/gcf
FINANCE=$GCF/COS513-Finance

$FINANCE/clusters.sh \
  $GCF/raw-data-20130401-20151021/ \
  $GCF/clusters-20130401-20150731 \
  20130401 \
  20150731

echo DONE WITH CLUSTERING $? | sendmail vyf@princeton.edu

$FINANCE/summary.sh \
  $GCF/raw-data-20130401-20151021/ \
  $GCFclusters-20130401-20150731 \
  $GCFCORRECT-summary-data-20130401-20151021

echo DONE WITH SUMMARIZING $? | sendmail vyf@princeton.edu

