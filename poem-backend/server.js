const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const port = 3000;

// 定义翻译数据
const translations = {
  '子曰：“学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？”': {
    title: "论语",
    author: "孔子",
    lines: ["子曰：“学而时习之，不亦说乎？", "有朋自远方来，不亦乐乎？", "人不知而不愠，不亦君子乎？"],
    modern_translation: "孔子说：“学习了并时常温习它，不也高兴吗？有同门师兄弟从远方来，不也快乐吗？人家不了解（我），（我）却不怨恨，不也是道德上有修养的人吗？”",
    modern_lines: ["孔子说：“学习了并时常温习它，不也高兴吗？", "有同门师兄弟从远方来，不也快乐吗？", "人家不了解（我），（我）却不怨恨，不也是道德上有修养的人吗？"]
  },
  '你好': {
    text: '你好，请问文言文或古诗词上有什么可以帮助到您？',
    translation: null
  }
};

app.use(cors());
app.use(bodyParser.json());

app.post('/api/login', (req, res) => {
  const { username, password } = req.body;
  if (username === 'student1' && password === '123456') {
    res.json({ token: 'sample_token' });
  } else {
    res.status(401).json({ message: 'Unauthorized' });
  }
});

app.post('/api/register', (req, res) => {
  const { username, password } = req.body;
  // 简单的注册逻辑
  res.json({ token: 'sample_token' });
});

app.post('/api/chat', (req, res) => {
  const { message } = req.body;
  const token = req.headers.authorization.split(' ')[1];
  if (token !== 'sample_token') {
    return res.status(401).json({ message: 'Unauthorized' });
  }

  if (message.includes('翻译')) {
    const key = message.replace('请帮忙翻译以下句子：', '').trim();
    const translation = translations[key] || {
      title: "未识别",
      author: "未知",
      lines: [key],
      modern_translation: "未识别到需要翻译的古诗词",
      modern_lines: ["未识别到需要翻译的古诗词"]
    };
    res.json({
      text: translation.title + " - " + translation.author,
      poem: {
        title: translation.title,
        author: translation.author,
        lines: translation.lines,
        modern_translation: translation.modern_translation,
        modern_lines: translation.modern_lines
      }
    });
  } else {
    const response = translations[message] || {
      text: '你好，请问文言文或古诗词上有什么可以帮助到您？',
      poem: null
    };
    res.json(response);
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});