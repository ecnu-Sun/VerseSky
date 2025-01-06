package com.example.demo.entity;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class PoemData {
    private Long id; // 主键
    private String title; // 诗的题目
    private String author; // 诗的作者
    private List<String> lines; // 诗句（List<String>）
    private String modernTranslation; // 现代文翻译
    private List<String> modernLines; // 现代文句子（List<String>）
    private Map<String, String> keywordsAnalysis; // 关键词分析（Map<String, String>）
    private String timestamp; // 插入时间
}
