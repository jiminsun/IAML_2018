# IAML_2018 Project 3

## Task

> 12초 길이의 wav 파일의 chord 를 맞추는 것  

* chord 는 0.125초당 한번 예측
* 한 파일당 총 96개의 chord 를 예측
* chord 의 종류
    * 12개의 음계에 대한 Major, minor chord 와 반주가 없을 때의 no chord 를 포함한 25가지
    
## Model requirements

* layer 2개 이상의 neural net

## Etc

* 데이터셋이 train 과 validation 으로 구분되어 있지 않아, 자유롭게 학습하면 된다.
* 주어진 데이터 모두를 사용하여 학습하는 것도 가능하다.
* 현재 뼈대코드는 main.py 를 실행시킬때마다 train, validation set 이 달라지게 되어 있음에 주의!


