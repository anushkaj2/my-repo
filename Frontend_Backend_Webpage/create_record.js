document.addEventListener('DOMContentLoaded', function() {
    function fetchPathogens() {
        fetch('backend.php?read=1')
            .then(response => response.json())
            .then(data => {
                const pathogenList = document.getElementById('pathogenList');
                pathogenList.innerHTML = '';
                data.forEach(pathogen => {
                    const pathogenItem = document.createElement('div');
                    pathogenItem.className = 'pathogenItem';
                    pathogenItem.innerHTML = `
                        <p>Pathogen ID: ${pathogen.pathogen_id}</p>
                        <p>Ideal Temperature: ${pathogen.ideal_temperature}</p>
                        <p>Incubation Period: ${pathogen.incubation_period}</p>
                        <p>Pathogenicity: ${pathogen.pathogenicity}</p>
                        <p>Transmission Mode: ${pathogen.transmission_mode}</p>
                    `;
                    pathogenList.appendChild(pathogenItem);
                });
            });
    }

    document.getElementById('pathogenForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(this);
        const pathogen_id = formData.get('pathogen_id');
        if (pathogen_id > 0) {
            formData.append('update', '1');
        } else {
            formData.append('create', '1');
        }
        fetch('backend.php', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
            fetchPathogens();
            document.getElementById('pathogenForm').reset();
            document.getElementById('createBtn').style.display = 'block';
            document.getElementById('updateBtn').style.display = 'none';
        });
    });

    window.editPathogen = function(pathogen_id, ideal_temperature, incubation_period, pathogenicity, transmission_mode) {
        document.getElementById('pathogenId').value = pathogen_id;
        document.getElementById('idealTemperature').value = ideal_temperature;
        document.getElementById('incubationPeriod').value = incubation_period;
        document.getElementById('pathogenicity').value = pathogenicity;
        document.getElementById('transmissionMode').value = transmission_mode;
        document.getElementById('createBtn').style.display = 'none';
        document.getElementById('updateBtn').style.display = 'block';
    }

    fetchPathogens();
});
