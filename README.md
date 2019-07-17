1. 输入如下命令。
   ```
   sudo apt-get install mysql-server
   ```
   安装期间会弹出界面，输入两次用户密码，就完成安装了。

2. mysql -u root -p，同样会要求输入用户密码。

3. 创建一个用户。这里的%代表可以在外网访问。
   ```
   CREATE USER 'username'@'%' IDENTIFIED BY 'password';
   ```

4. 如果刚才创建的是超级用户，那么可以给它所有数据库的所有权限。
   ```
   grant all privileges on *.* to username;
   ```

5. 最重要的一步，vim /etc/mysql/mysql.conf.d/mysqld.cnf，把bind-address = 127.0.0.1注释掉，然后输入如下命令将mysql重启，这样数据库才能够远程访问。
   ```
   service mysqld restart
   ```
