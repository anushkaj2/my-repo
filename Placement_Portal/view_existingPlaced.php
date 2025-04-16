<?php
header('Content-Type: application/json');

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

// SQL query to check if student exists in Placed table
$sql = "SELECT * FROM Placed WHERE Registration_No = '$registrationNo'";
$result = $conn->query($sql);

$response = array('exists' => $result->num_rows > 0);

echo json_encode($response);

$conn->close();
?>
