-- ============================================================
-- Agri-Agent 初始化脚本（MySQL 8.0）
-- 执行者：实例管理员账号（需要 CREATE USER / GRANT 权限）
-- 注意：这是与他人共用的实例，本脚本只操作 agri_agent 库和 agri 账号，
--       不触碰、不修改任何既有库与账号。
-- ============================================================

-- ---- 1) 专用库 ----
-- 显式指定 charset/collation：即使实例默认是 latin1，也要保证本库是 utf8mb4，
-- 否则建表时会继承实例默认值，存中文直接乱码或报错。
-- utf8mb4_0900_ai_ci 是 8.0 专有 collation，5.7 上会报错（本项目已确认 8.0）。
CREATE DATABASE IF NOT EXISTS agri_agent
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

-- ---- 2) 确认 GRANT 的 host 部分该写什么 ----
-- OFF（默认）：MySQL 会把 127.0.0.1 反解析成 localhost → 需要 'agri'@'localhost'
-- ON        ：不做反解析，host 就是 IP            → 需要 'agri'@'127.0.0.1'
-- 这是 "Access denied for user 'agri'@'localhost'" 最常见的来源。
-- 下面两个 host 都建了，成本为零，省掉一轮排查。
SELECT @@skip_name_resolve;

-- ---- 3) 专用账号 ----
-- 密码请与其它项目的账号不同。注意 IF NOT EXISTS 在账号已存在时会静默跳过、
-- 不会更新密码；若要改密码请用 ALTER USER ... IDENTIFIED BY '新密码';
-- 【本文件会进版本库】故此处只留占位符：执行前请把 <YOUR_DB_PASSWORD> 换成
-- 你的强密码，并同步更新 .env 的 DATABASE_URL（.env 不入库）。
CREATE USER IF NOT EXISTS 'agri'@'localhost' IDENTIFIED BY '<YOUR_DB_PASSWORD>';
CREATE USER IF NOT EXISTS 'agri'@'127.0.0.1' IDENTIFIED BY '<YOUR_DB_PASSWORD>';

-- ---- 4) 授权：仅到 agri_agent.*，且刻意不给 DROP ----
-- CREATE 权限是为了让应用的 init_db() 能在首次启动时自动建表。
-- 若你更想要最小权限，去掉 CREATE，然后手工执行下面第 6 节的 DDL：
-- create_all() 会先反射现有表，发现已存在就跳过，不会发 DDL，因此不会因缺权限而失败。
-- 不给 DROP 是有意的：将来万一带 bug，删表也需要人工介入。
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX
  ON agri_agent.* TO 'agri'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX
  ON agri_agent.* TO 'agri'@'127.0.0.1';
FLUSH PRIVILEGES;

-- ---- 5) 验证隔离生效 ----
-- 期望只看到 agri_agent.* 上的那几条，不应出现任何通配或别的库名。
SHOW GRANTS FOR 'agri'@'localhost';

/*
CREATE TABLE IF NOT EXISTS agri_agent.users (
  id            INT          NOT NULL AUTO_INCREMENT,
  username      VARCHAR(64)  NOT NULL,
  password_hash VARCHAR(128) NOT NULL,
  created_at    DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_users_username (username)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;
*/