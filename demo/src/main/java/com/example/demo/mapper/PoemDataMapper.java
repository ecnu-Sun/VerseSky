package com.example.demo.mapper;

import com.example.demo.entity.PoemData;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;


@Mapper
public interface PoemDataMapper {

    /**
     * 插入诗歌数据（只保存 title、author 和 timestamp）
     * @param poemData 诗歌数据对象
     */

    void insertPoemData(PoemData poemData);
}
