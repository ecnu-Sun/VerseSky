package com.example.demo.controller;

import com.example.demo.entity.PoemData;
import com.example.demo.service.PoemDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.sql.CallableStatement;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/action")
public class PoemDataController {

    @Autowired
    private PoemDataService poemDataService;

    /**
     * 保存诗歌数据
     * @param requestData 包含 action 和 poem_data 的请求数据
     * @return 保存后的完整诗歌数据
     */
    @PostMapping("/updateview")
    public PoemData savePoemData(@RequestBody PoemData requestData) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        requestData.setTimestamp(timestamp); // 填充 timestamp 字段
        // 调用 Service 层保存数据
        return poemDataService.savePoemData(requestData);
    }
}
