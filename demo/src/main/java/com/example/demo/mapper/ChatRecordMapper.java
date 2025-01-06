package com.example.demo.mapper;

import com.example.demo.entity.ChatRecord;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ChatRecordMapper {
    void insertChatRecord(@Param("record") ChatRecord record);
}
