tee /etc/yum.repos.d/neuron.repo > /dev/null << EOF 
[neuron] 
name=Neuron YUM Repository 
baseurl=https://yum.repos.neuron.amazonaws.com 
enabled=1 
metadata_expire=0
EOF
rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB 
dnf update -y 
dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y 
dnf install clang llvm-devel -y 
dnf install aws-neuronx-dkms-2.* -y 
dnf install aws-neuronx-collectives-2.* -y 
dnf install aws-neuronx-runtime-lib-2.* -y 
dnf install aws-neuronx-tools-2.* -y 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y