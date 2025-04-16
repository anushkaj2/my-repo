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

// Get the registration number from POST data
$registrationNo = $conn->real_escape_string($_POST['registrationNo']);

// SQL query to delete the student from Placed table
$sql = "DELETE FROM Placed WHERE Registration_No = '$registrationNo'";

if ($conn->query($sql) === TRUE) {
    echo "Student removed from Placed table successfully.";
} else {
    echo "Error: " . $conn->error;
}

$conn->close();
?>
