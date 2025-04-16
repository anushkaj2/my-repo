<?php
header('Content-Type: text/html; charset=utf-8');

// Database connection
$servername = "localhost";
$username = "IBAB_Placements";
$password = "placementportal@24";
$dbname = "Placements";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get the registration number, company, position and salary from POST data
$registrationNo = $conn->real_escape_string($_POST['registrationNo']);
$company = $conn->real_escape_string($_POST['company']);
$position = $conn->real_escape_string($_POST['position']);
$salary = $conn->real_escape_string($_POST['salary']);

// Fetch student name and program
$sql = "SELECT `Name`, Program FROM Student_info WHERE Registration_No = '$registrationNo'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    $name = $row['Name'];
    $program = $row['Program'];

    // Insert into Placed table
    $sql = "INSERT INTO Placed (Registration_No, `Name`, Program, Company, Position, Salary) VALUES ('$registrationNo', '$name', '$program', '$company', '$position', '$salary')";
    
    if ($conn->query($sql) === TRUE) {
        echo "Student added to Placed table successfully.";
    } else {
        echo "Error: " . $conn->error;
    }
} else {
    echo "Student not found.";
}

$conn->close();
?>
