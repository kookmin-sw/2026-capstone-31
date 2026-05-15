import torch
import torch.nn as nn

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        """
        Args:
            input_size: 랜드마크 좌표 수 (21개 점 * x,y,z = 63)
            hidden_size: LSTM 은닉층의 노드 수
            num_layers: LSTM 층의 개수
            num_classes: 인식할 수어 단어/동작의 개수
        """
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        # batch_first=True는 입력 데이터의 형태가 (batch, sequence, feature)임을 의미합니다.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected 레이어 (마지막 출력층)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 초기 Hidden state와 Cell state 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 통과
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시점(Time step)의 결과값만 가져와서 분류기에 입력
        out = self.fc(out[:, -1, :])
        
        return out