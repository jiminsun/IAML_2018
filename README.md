# IAML_2018
*Industrial Application of Machine Learning* , Fall 2018, Seoul National University.

## Project 1
### Genre classification   
* Requirements
    * 30초 길이의 노래가 주어졌을 때 해당 곡의 장르를 맞추기
    * CNN 계열의 모델 사용

## Project 2
### Hit prediction   
* Requirements
    * 30초 길이의 노래가 주어졌을 때 해당 곡의 재생 횟수 맞추기
    * 5개의 수준으로 나누어 classification 문제로 접근
    * RNN 계열의 모델 사용

## Project 3
### Chord prediction  
* Requirements
    * 12초 길이의 wav 파일의 chord 맞추기 
* Results
    * Implemented the CRNN structure suggested in [Choi. et al (2017)](https://ieeexplore.ieee.org/abstract/document/7952585) with some modifications to address to this specific task.
    * The model showed 98% accuracy on the training set and 93% accuracy on the validation set, on average.
    * The performance could be improved via other models such as the [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need), but this haven't been tried yet.
    * The raw music files aren't provided in this repository.

## Project 4
### Music Generation
TBA

