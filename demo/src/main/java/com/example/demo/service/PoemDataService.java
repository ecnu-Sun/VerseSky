package com.example.demo.service;

import com.example.demo.mapper.PoemDataMapper;
import com.example.demo.entity.PoemData;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.awt.*;

@Service
public class PoemDataService {

    @Autowired
    private PoemDataMapper poemDataMapper;

    /**
     * 保存诗歌数据
     * @param poemData 包含诗歌数据的对象
     * @return 保存后的完整诗歌数据
     */
    public PoemData savePoemData(PoemData poemData) {
        // 保存部分数据（title、author、timestamp）到数据库
        poemDataMapper.insertPoemData(poemData);

        // 返回完整的 PoemData 数据
        return poemData;
    }
}