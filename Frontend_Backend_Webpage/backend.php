<?php
// Database connection
$servername = "localhost";
$username = "archive_admin";
$password = "mobilecharger";
$database = "epidemic_archive";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Create operation
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["create"])) {
    $pathogen_id = $_POST["pathogen_id"];
    $ideal_temperature = $_POST["ideal_temperature"];
    $incubation_period = $_POST["incubation_period"];
    $pathogenicity = $_POST["pathogenicity"];
    $transmission_mode = $_POST["transmission_mode"];

    $sql = "INSERT INTO pathogen (pathogen_id, ideal_temperature, incubation_period, pathogenicity, transmission_mode) VALUES ('$pathogen_id', '$ideal_temperature', '$incubation_period', '$pathogenicity', '$transmission_mode')";

    if ($conn->query($sql) === TRUE) {
        echo "New record created successfully";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

// Read operation
if ($_SERVER["REQUEST_METHOD"] == "GET" && isset($_GET["read"])) {
    $sql = "SELECT * FROM pathogen";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $rows = array();
        while ($row = $result->fetch_assoc()) {
            $rows[] = $row;
        }
        echo json_encode($rows);
    } else {
        echo "0 results";
    }
}

// Update operation
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["update"])) {
    $pathogen_id = $_POST["pathogen_id"];
    $ideal_temperature = $_POST["ideal_temperature"];
    $incubation_period = $_POST["incubation_period"];
    $pathogenicity = $_POST["pathogenicity"];
    $transmission_mode = $_POST["transmission_mode"];

    $sql = "UPDATE pathogen SET ideal_temperature='$ideal_temperature', incubation_period='$incubation_period', pathogenicity='$pathogenicity', transmission_mode='$transmission_mode' WHERE pathogen_id=$pathogen_id";

    if ($conn->query($sql) === TRUE) {
        echo "Record updated successfully";
    } else {
        echo "Error updating record: " . $conn->error;
    }
}

// Delete operation
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["delete"])) {
    $pathogen_id = $_POST["pathogen_id"];

    $sql = "DELETE FROM pathogen WHERE pathogen_id='$pathogen_id'";
    if ($conn->query($sql) === TRUE) {
        echo "Record deleted successfully";
    } else {
        echo "Error deleting record: " . $conn->error;
    }
}

$conn->close();
?>
