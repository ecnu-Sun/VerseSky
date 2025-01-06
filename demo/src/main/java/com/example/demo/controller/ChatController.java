package com.example.demo.controller;

import com.example.demo.service.ChatRecordService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    @Autowired
    private ChatRecordService chatRecordService;

    private final String PYTHON_API_URL = "http://localhost:8000/process";

    @PostMapping("/translate")
    public Map<String, Object> translatePoem(@RequestBody Map<String, String> request) {
        String originalText = request.get("input_text");
        if (originalText == null || originalText.isEmpty()) {
            throw new IllegalArgumentException("输入文本不能为空");
        }

        // 调用 Python 后端
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");

        Map<String, String> body = new HashMap<>();
        body.put("input_text", originalText);

        HttpEntity<Map<String, String>> entity = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = restTemplate.exchange(PYTHON_API_URL, HttpMethod.POST, entity, Map.class);

        // 获取翻译结果
        Map<String, Object> pythonResponse = response.getBody();
        if (pythonResponse == null || !pythonResponse.containsKey("data")) {
            throw new RuntimeException("翻译失败");
        }


        // 提取 data 对象
        Map<String, Object> data = (Map<String, Object>) pythonResponse.get("data");
        if (!data.containsKey("output")) {
            throw new RuntimeException("翻译结果中缺少 output 字段");
        }

        String translatedText = data.get("output").toString();
        // 保存聊天记录
        chatRecordService.saveChatRecord(originalText, translatedText);

        // 返回给前端
        Map<String, Object> result = new HashMap<>();
        result.put("translatedText", translatedText);
        return result;
    }
}
