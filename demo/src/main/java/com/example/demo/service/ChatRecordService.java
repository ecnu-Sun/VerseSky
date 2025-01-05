package com.example.demo.service;

import com.example.demo.entity.ChatRecord;
import com.example.demo.mapper.ChatRecordMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Service
public class ChatRecordService {

    @Autowired
    private ChatRecordMapper chatRecordMapper;

    public void saveChatRecord(String originalText, String translatedText) {
        ChatRecord record = new ChatRecord();
        record.setOriginalText(originalText);
        record.setTranslatedText(translatedText);
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        record.setTimestamp(timestamp);
        chatRecordMapper.insertChatRecord(record);
    }
}
