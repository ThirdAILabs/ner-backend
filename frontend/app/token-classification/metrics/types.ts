export interface TokenFeedback {
  object: string;
  tokens: string[];
  tags: string[];
}

export const mockFeedbackData: TokenFeedback[] = [
  {
    object:
      '1e82d3fd-c36a-40a1-a3af-81da9d464fd4/deepseek sparse attention long context.pdf',
    tokens: [
      'Native Sparse Attention: Hardware-Aligned and Natively Trainable ',
      'Sparse',
      ' Attention Jingyang Yuan∗1,2, ',
      'Huazuo',
      ' ',
      'Gao1,',
      ' ',
      'Damai',
      ' Dai1, ',
      'Junyu',
      ' ',
      'Luo2,',
      ' ',
      'Liang',
      ' Zhao1, ',
      'Zhengyan',
      ' Zhang1, ',
      'Zhenda',
      ' ',
      'Xie1,'
      // ... Add more tokens as needed (shortened for brevity)
    ],
    tags: [
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME',
      'O',
      'NAME'
      // ... Add more tags as needed (shortened for brevity)
    ]
  }
];
