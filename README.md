<!DOCTYPE html>
<html>
<head>

</head>
<body>

<h2>Installation guide</h2>  
<p>This is how the framework can be downloaded and configured to run the experiments</p>
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to "pull Docker Image" from Docker Hub: <code>docker pull shefai/intent_aware_recomm_systems</code>
  <li>Clone the GitHub repository by using the link: <code>https://github.com/Faisalse/KGIN.git</code>
  <li>Move into the <b>KGIN</b> directory</li>
  
  <li>Run the command to mount the current directory <i>KGIN</i> to the docker container named as <i>KGIN_container</i>: <code>docker run --name KGIN_container  -it -v "$(pwd):/KGIN" -it shefai/intent_aware_recomm_systems</code>. If you have the support of CUDA-capable GPUs then run the following command to attach GPUs with the container: <code>docker run --name KGIN_container  -it --gpus all -v "$(pwd):/KGIN" -it shefai/intent_aware_recomm_systems</code></li> 
<li>Finally, follow the given instructions to run the experiments for KGIN and baseline models </li>
</ul> 

<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/KGIN.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>KGIN</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name KGIN python=3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate KGIN</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_cpu.txt</code>. However, if you have support of CUDA-capable GPUs, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt</code></li>
  </ul>
</p>



<h4>Follow these steps to reproduce the results for KIGN and baseline models</h4>
<ul>
<li>Run this command to reproduce the experiments for the KGIN and baseline models on the lastFm dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset lastFm</code>  </li>
<li>Run this command to reproduce the experiments for the KGIN and baseline models on the alibabaFashion dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset alibabaFashion</code>  </li>
<li>Run this command to reproduce the experiments for the KGIN and baseline models on the amazonBook dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset amazonBook</code>  </li>
</ul>






</body>
</html>  

