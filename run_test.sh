# 设置默认值
testmode="internode"
profiling=""

for para in $*
do
    if [[ $para == --testmode* ]];then
        testmode=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
echo "CURRENT_DIR = ${CURRENT_DIR}"
TEST_DIR=${CURRENT_DIR}/tests_mpi

LAUNCH_WITH_BINDING=${TEST_DIR}/launch_with_binding.sh # Please adjust the variables based on the actual NET being used
DTK_ENV="/opt/dtk/env.sh" # where env.sh of dtk
TEST_ENV=${TEST_DIR}/test_env.sh

#######################################################################################
# Those variables no need to modify
# HOSTFILE="hostfile_$(basename "$0" | sed -E 's/^run_(.+)\.sh$/\1/')"
HOSTFILE="${TEST_DIR}/hostfile"
GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l) * 8))
HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="1234"
echo "HOST=${HOST}, PORT=${PORT}, GPUS=${GPUS}"

# Runs aibenchmark model
source ${TEST_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${TEST_ENV} && \
                    export TEST_DIR=${TEST_DIR} && \
                    ${TEST_DIR}/test_start.sh \
                    ${HOST} \
                    ${PORT} \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --testmode=${testmode} \
                    --profiling=${profiling}
                    "
		            # > log-${testmode}-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log 2>&1
wait


## 修改 hostfile，指定测试的模式
# ./run_test.sh --testmode=intranode 
# ./run_test.sh --testmode=internode 
# ./run_test.sh --testmode=lowlatency 
