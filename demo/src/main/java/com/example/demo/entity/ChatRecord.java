package com.example.demo.entity;

import lombok.Data;

@Data
public class ChatRecord {
    private Long id;
    private String originalText;
    private String translatedText;
    private String timestamp;

}
