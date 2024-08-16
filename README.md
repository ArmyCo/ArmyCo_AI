# ReCo_AI

**requirements**
* torch=2.2.2+cu121
* transformers=4.40.0

## How to use
* TourEmbedding : 관광지 정보를 기반으로 관광지의 feature를 만드는 embedding 모델
  - location_embedding.pth : 위치 정보 기반으로 학습된 임베딩 테이블

* SimBasedSys : 벡터의 유사도 기반으로 관광지를 추천해주는 모델
  - tour_embedding.pth : TourEmbedding에서 구한 관광지 임베딩 테이블
