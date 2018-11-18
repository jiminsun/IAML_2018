Project2.md <br>
updated : 2018-11-18-12:39 - 파일 이름 변경
<!-- 수정시 아래 이어서 updated 날짜 시간 & 수정사항 추가 -->

# IAML Project 2
Requirement : 음악 파일과 track_listen()을 RNN을 사용해 학습해서 (5단계) test파일에 대해 track_listen 등급 맞추기

# TODO


<전처리>
* features, dataloader 쪽 함수 새로만들기 -> data 불러오는 순서 등 pipeline

<slicing 함수>
- sample number 
- window size(of one piece)
- ramdomize
- padding

<feature뽑는 함수>


<단위feature의 예측률>
- 장르별로, slice된 것 중 분류에 영향을 주었던 패턴 찾기 (e.g 앞쪽이 잘 나오는지 등)
- 장르별로, 파일에서 만들어진 interval에서 loss의 차이

<테스트해볼것>
- 장르별 track_listen 분포 그려보기 표/histogram/pie chart등
- 
- mfcc threshold에 대해 sr바꿔보기
- threshold에 짤리는 음악
    - padding
    - 이어붙이기
    - 뒤집어서붙이기
    - 버리기 (원래방법)
- features
    - mfcc
    - melspec
    - cqt


<메인함수 순서>
- 데이터 피클링(없으면 피클링 있으면 스킵)
- slicing유무/방법
- 데이터로딩
- 모델학습
    - early stopping
    - validation error
- validation accuracy
- ensemble
- ensemble accuracy


----

전처리
- 어떻게 자를지를 함수안에 option
모델
- training
- LSTM으로 시작 (y, logit을 output)
평가
- accuracy
- visualization (loss, accuracy)



----
filezilla
