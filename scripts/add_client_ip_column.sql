-- ============================================
-- Script: Add client_ip column to log table
-- FaceLibre - Face Recognition API
-- Compatible with MySQL 5.7 and earlier
-- ============================================

-- Check if column exists, if not add it
SET @column_exists = (
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() 
    AND TABLE_NAME = 'tc_face_recognition_log' 
    AND COLUMN_NAME = 'client_ip'
);

SET @sql = IF(@column_exists = 0, 
    'ALTER TABLE tc_face_recognition_log ADD COLUMN client_ip VARCHAR(45) AFTER timestamp',
    'SELECT "Column client_ip already exists" AS message'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Verify the change
DESCRIBE tc_face_recognition_log;
