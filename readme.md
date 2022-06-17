# Group Project
* Members: Xi Wang, Binhao Yan, Zhongyi Wang, Yang Zhang, Duanmiao Si
* Puzzles choice: 基于胸部X-ray成像的肺炎诊断  



## Requirements

```
name: torch
channels:
  - pytorch
  - dglteam
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=4.5=1_gnu
  - _pytorch_select=0.1=cpu_0
  - absl-py=0.13.0=py38h06a4308_0
  - aiohttp=3.7.4.post0=py38h7f8727e_2
  - async-timeout=3.0.1=py38h06a4308_0
  - attrs=21.2.0=pyhd3eb1b0_0
  - backcall=0.2.0=pyhd3eb1b0_0
  - blas=1.0=mkl
  - blinker=1.4=py38h06a4308_0
  - boost=1.74.0=py38hc10631b_3
  - boost-cpp=1.74.0=h9359b55_0
  - boto3=1.18.21=pyhd3eb1b0_0
  - botocore=1.21.41=pyhd3eb1b0_1
  - bottleneck=1.3.2=py38heb32a55_1
  - brotli=1.0.9=he6710b0_2
  - brotlipy=0.7.0=py38h27cfd23_1003
  - bzip2=1.0.8=h7b6447c_0
  - c-ares=1.17.1=h27cfd23_0
  - ca-certificates=2021.10.8=ha878542_0
  - cachetools=4.2.2=pyhd3eb1b0_0
  - cairo=1.16.0=hf32fb01_1
  - certifi=2021.10.8=py38h578d9bd_1
  - cffi=1.14.6=py38h400218f_0
  - chardet=4.0.0=py38h06a4308_1003
  - charset-normalizer=2.0.4=pyhd3eb1b0_0
  - click=8.0.3=pyhd3eb1b0_0
  - coverage=5.5=py38h27cfd23_2
  - cryptography=3.4.8=py38hd23ed53_0
  - cudatoolkit=11.3.1=h2bc3f7f_2
  - cycler=0.10.0=py38_0
  - cython=0.29.24=py38hdbfa776_0
  - dataclasses=0.8=pyh6d0b6a4_7
  - decorator=5.1.0=pyhd3eb1b0_0
  - dgl-cuda11.1=0.7.1=py38_0
  - et_xmlfile=1.1.0=py38h06a4308_0
  - ffmpeg=4.3=hf484d3e_0
  - fontconfig=2.13.1=h6c09931_0
  - fonttools=4.25.0=pyhd3eb1b0_0
  - freetype=2.11.0=h70c0345_0
  - gensim=3.8.3=py38h2531618_2
  - giflib=5.2.1=h7b6447c_0
  - glib=2.69.1=h5202010_0
  - gmp=6.2.1=h2531618_2
  - gnutls=3.6.15=he1e5248_0
  - google-api-core=1.25.1=pyhd3eb1b0_0
  - google-auth=1.33.0=pyhd3eb1b0_0
  - google-auth-oauthlib=0.4.1=py_2
  - google-cloud-core=1.7.1=pyhd3eb1b0_0
  - google-cloud-storage=1.41.0=pyhd3eb1b0_0
  - google-crc32c=1.1.2=py38h27cfd23_0
  - google-resumable-media=1.3.1=pyhd3eb1b0_1
  - googleapis-common-protos=1.53.0=py38h06a4308_0
  - greenlet=1.1.1=py38h295c915_0
  - grpcio=1.36.1=py38h2157cd5_1
  - icu=67.1=he1b5a44_0
  - idna=3.2=pyhd3eb1b0_0
  - importlib-metadata=4.8.1=py38h06a4308_0
  - intel-openmp=2019.4=243
  - ipython=7.29.0=py38hb070fc8_0
  - jedi=0.18.0=py38h06a4308_1
  - jmespath=0.10.0=pyhd3eb1b0_0
  - joblib=1.1.0=pyhd3eb1b0_0
  - jpeg=9d=h7f8727e_0
  - kiwisolver=1.3.1=py38h2531618_0
  - lame=3.100=h7b6447c_0
  - lcms2=2.12=h3be6417_0
  - ld_impl_linux-64=2.35.1=h7274673_9
  - libcrc32c=1.1.1=he6710b0_2
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.3.0=h5101ec6_17
  - libgfortran-ng=7.5.0=ha8ba4b0_17
  - libgfortran4=7.5.0=ha8ba4b0_17
  - libgomp=9.3.0=h5101ec6_17
  - libiconv=1.16=h516909a_0
  - libidn2=2.3.2=h7f8727e_0
  - libmklml=2019.0.5=0
  - libpng=1.6.37=hbc83047_0
  - libprotobuf=3.17.2=h4ff587b_1
  - libstdcxx-ng=9.3.0=hd4cf53a_17
  - libtasn1=4.16.0=h27cfd23_0
  - libtiff=4.2.0=h85742a9_0
  - libunistring=0.9.10=h27cfd23_0
  - libuuid=1.0.3=h7f8727e_2
  - libuv=1.40.0=h7b6447c_0
  - libwebp=1.2.0=h89dd481_0
  - libwebp-base=1.2.0=h27cfd23_0
  - libxcb=1.14=h7b6447c_0
  - libxml2=2.9.10=h68273f3_2
  - lz4-c=1.9.3=h295c915_1
  - markdown=3.3.4=py38h06a4308_0
  - matplotlib-base=3.4.3=py38hbbc1b5f_0
  - matplotlib-inline=0.1.2=pyhd3eb1b0_2
  - mkl=2020.2=256
  - mkl-service=2.3.0=py38he904b0f_0
  - mkl_fft=1.3.0=py38h54f3939_0
  - mkl_random=1.1.1=py38h0573a6f_0
  - multidict=5.1.0=py38h27cfd23_2
  - munch=2.5.0=pyhd3eb1b0_0
  - munkres=1.1.4=py_0
  - ncurses=6.3=h7f8727e_2
  - nettle=3.7.3=hbbd107a_1
  - networkx=2.6.3=pyhd3eb1b0_0
  - ninja=1.10.2=py38hd09550d_3
  - numexpr=2.7.3=py38hb2eb853_0
  - numpy=1.19.2=py38h54aff64_0
  - numpy-base=1.19.2=py38hfa32c7d_0
  - oauthlib=3.1.1=pyhd3eb1b0_0
  - olefile=0.46=pyhd3eb1b0_0
  - openbabel=3.1.1=py38hf4b5c11_1
  - openh264=2.1.0=hd408876_0
  - openpyxl=3.0.9=pyhd3eb1b0_0
  - openssl=1.1.1n=h7f8727e_0
  - pandas=1.3.3=py38h8c16a72_0
  - parso=0.8.2=pyhd3eb1b0_0
  - pcre=8.45=h295c915_0
  - pexpect=4.8.0=pyhd3eb1b0_3
  - pickleshare=0.7.5=pyhd3eb1b0_1003
  - pillow=8.4.0=py38h5aabda8_0
  - pip=21.2.4=py38h06a4308_0
  - pixman=0.40.0=h7f8727e_1
  - prompt-toolkit=3.0.20=pyhd3eb1b0_0
  - protobuf=3.17.2=py38h295c915_0
  - ptyprocess=0.7.0=pyhd3eb1b0_2
  - pyasn1=0.4.8=pyhd3eb1b0_0
  - pyasn1-modules=0.2.8=py_0
  - pycairo=1.19.1=py38h708ec4a_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pygments=2.10.0=pyhd3eb1b0_0
  - pyjwt=2.1.0=py38h06a4308_0
  - pyopenssl=21.0.0=pyhd3eb1b0_1
  - pyparsing=3.0.4=pyhd3eb1b0_0
  - pysocks=1.7.1=py38h06a4308_0
  - python=3.8.12=h12debd9_0
  - python-dateutil=2.8.2=pyhd3eb1b0_0
  - python_abi=3.8=2_cp38
  - pytorch=1.10.0=py3.8_cuda11.3_cudnn8.2.0_0
  - pytorch-mutex=1.0=cuda
  - pytz=2021.3=pyhd3eb1b0_0
  - rdkit=2020.09.5=py38h2bca085_0
  - readline=8.1=h27cfd23_0
  - reportlab=3.5.67=py38hfdd840d_1
  - requests=2.26.0=pyhd3eb1b0_0
  - requests-oauthlib=1.3.0=py_0
  - rsa=4.7.2=pyhd3eb1b0_1
  - s3transfer=0.5.0=pyhd3eb1b0_0
  - scikit-learn=1.0.1=py38h51133e4_0
  - scipy=1.6.2=py38h91f5cce_0
  - setuptools=58.0.4=py38h06a4308_0
  - six=1.16.0=pyhd3eb1b0_0
  - smart_open=5.1.0=pyhd3eb1b0_0
  - sqlalchemy=1.4.22=py38h7f8727e_0
  - sqlite=3.36.0=hc218d9a_0
  - tensorboard=2.6.0=py_1
  - tensorboard-data-server=0.6.0=py38hca6d32c_0
  - tensorboard-plugin-wit=1.6.0=py_0
  - threadpoolctl=2.2.0=pyh0d69192_0
  - tk=8.6.11=h1ccaba5_0
  - torchaudio=0.10.0=py38_cu113
  - torchvision=0.11.1=py38_cu113
  - tornado=6.1=py38h27cfd23_0
  - tqdm=4.62.3=pyhd3eb1b0_1
  - traitlets=5.1.1=pyhd3eb1b0_0
  - urllib3=1.26.7=pyhd3eb1b0_0
  - wcwidth=0.2.5=pyhd3eb1b0_0
  - werkzeug=2.0.2=pyhd3eb1b0_0
  - wheel=0.37.0=pyhd3eb1b0_1
  - xz=5.2.5=h7b6447c_0
  - yarl=1.5.1=py38h7b6447c_0
  - zipp=3.6.0=pyhd3eb1b0_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.4.9=haebb681_0
  - pip:
    - argon2-cffi==21.3.0
    - argon2-cffi-bindings==21.2.0
    - beautifulsoup4==4.10.0
    - bleach==4.1.0
    - cloudpickle==2.0.0
    - debugpy==1.5.1
    - defusedxml==0.7.1
    - entrypoints==0.4
    - fsspec==2022.3.0
    - future==0.18.2
    - hyperopt==0.2.7
    - importlib-resources==5.4.0
    - ipykernel==6.9.1
    - ipython-genutils==0.2.0
    - ipywidgets==7.6.5
    - jinja2==3.0.3
    - jsonschema==4.4.0
    - jupyter==1.0.0
    - jupyter-client==7.1.2
    - jupyter-console==6.4.3
    - jupyter-core==4.9.2
    - jupyterlab-pygments==0.1.2
    - jupyterlab-widgets==1.0.2
    - llvmlite==0.37.0
    - markupsafe==2.1.0
    - mistune==0.8.4
    - nbclient==0.5.13
    - nbconvert==6.4.4
    - nbformat==5.2.0
    - nest-asyncio==1.5.4
    - notebook==6.4.8
    - numba==0.54.1
    - opencv-python==4.5.4.58
    - packaging==21.3
    - pandocfilters==1.5.0
    - prometheus-client==0.13.1
    - py4j==0.10.9.3
    - pydeprecate==0.3.2
    - pynndescent==0.5.5
    - pyrsistent==0.18.1
    - pytorch-lightning==1.6.3
    - pyyaml==6.0
    - pyzmq==22.3.0
    - qtconsole==5.2.2
    - qtpy==2.0.1
    - seaborn==0.11.2
    - send2trash==1.8.0
    - soupsieve==2.3.1
    - terminado==0.13.3
    - testpath==0.6.0
    - torchmetrics==0.8.2
    - typing-extensions==4.2.0
    - umap-learn==0.5.2
    - webencodings==0.5.1
    - widgetsnbextension==3.5.2
    - xgboost==1.5.2
prefix: /home/dqw_cw/miniconda3/envs/torch
```



## How to use
本项目集成了LSTM，DNN，ResNet， Vgg，SqueezeNet等主流神经网络用于解决此分类问题。同时为了加快训练速度使用了DDP，AMP混合精度加速等方法。

### Model List
* resnet18, resnet34
* vgg16, vgg34
* squeezeNet_0, squeezeNet_1
* Linear Layer
* LSTM+ResNet18

### Optimizer List
* Adam
* SGD

### Mixed Precision Boosting
* Amp O1
* FP16/32

### Parallel Boosting
* DataParallel
* Distributed DataParallel


### train step

```
python3 main.py --train True --model_type=resnet18 --data_dir=<Your data dir> --nums_gpu=4
```
All the result will be loaded into lighting_logs folder.

### test step
```
python3 test.py --model_type <Your model> --model_path <Your Path>
```
