-- Create a new user and set password
CREATE USER 'IBAB_Placements'@'%' IDENTIFIED BY 'placementportal@24';

-- Grant all privileges on 'Placements' to 'IBAB_Placements'
GRANT ALL PRIVILEGES ON Placements.* TO 'IBAB_Placements'@'%';

-- Flush privileges to apply changes
FLUSH PRIVILEGES;

-- used to log in into the database with created username and password
mysql -u IBAB_Placements -p;

-- Create a new database
CREATE DATABASE Placements;

use Placements;

-- Creating tables

create table Student_info
(Registration_No varchar(20) not null,
 Name varchar(100) not null,
 Email varchar(100) not null,
 DOB DATE not null,
 Program varchar(50) not null, 
 Area_Of_Interest varchar(250) not null,
 Preferred_Location varchar(150) not null, 
 CGPA float not null, 
 CV LONGBLOB not null, 
 primary key(Registration_No));

create table Placed
(Registration_No varchar(20) not null,
 Name varchar(100) not null,
 Program varchar(50) not null,
 Company varchar(200) not null,
 Position varchar(200) not null,
 Salary varchar(200),
 primary key(Registration_No));







 

