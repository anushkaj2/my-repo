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

// Get the registration number, new company, new position and new salary from POST data
$registrationNo = $conn->real_escape_string($_POST['registrationNo']);
$newCompany = $conn->real_escape_string($_POST['company']);
$newPosition = $conn->real_escape_string($_POST['position']);
$newSalary = $conn->real_escape_string($_POST['salary']);

// SQL query to update the student record in Placed table
$sql = "UPDATE Placed SET Company = '$newCompany', Position = '$newPosition', Salary = '$newSalary' WHERE Registration_No = '$registrationNo'";

if ($conn->query($sql) === TRUE) {
    echo "Student record updated successfully.";
} else {
    echo "Error: " . $conn->error;
}

$conn->close();
?>
