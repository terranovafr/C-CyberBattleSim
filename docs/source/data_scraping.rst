.. _data-scraping:

Data Scraping
=============

The goal of this environment is to realistically simulate networks by integrating **real-world services and vulnerabilities**. To support this, we developed a **data scraping module** that collects:

- **Live service statistics** from Shodan (e.g., commonly deployed services in a certain geographic area and their versions).
- **Associated vulnerabilities** to the service versions using the National Vulnerability Database (NVD).

This ensures realism while preserving **anonymity** and **privacy**—no actual devices are targeted or identified.
Shodan is used purely for statistical purposes, preserving anonymization and privacy by ensuring that no vulnerabilities are sourced directly from real-world devices.
The data scraping module builds a structured database of tuples in the form: *(service, version, vulnerabilities, device count)*.
This database is later used in the `Scenario Generation <scenario_generation.html>`_ step to assign services and vulnerabilities to nodes based on real-world distributions.

API Key Setup
-------------

Before scraping, make sure to configure your API keys by creating a file at: ``cyberbattle/data_generation/config/auth.yaml``. The file should have the following structure:

.. code-block:: yaml

    shodan:
        key: YOUR_SHODAN_API_KEY

    nvd:
        key: YOUR_NVD_API_KEY

    huggingface:
        key: YOUR_HUGGINGFACE_API_KEY

Scraping Command
-----------------

Once the API keys are configured, run the scraping pipeline using a single command:

.. code-block:: bash

    cd cyberbattle/data_generation

    python scrape_data.py \
        --query SHODAN_QUERY \
        --num_products NUM_PRODUCTS \
        --num_versions NUM_VERSIONS_PER_PRODUCT \
        --nlp_extractors NLP_EXTRACTORS_LIST

This will populate several files inside the logs folder: ``cyberbattle/data/scrape_samples/``.

The following steps are executed in sequence and each step's output is saved in a different file of the logs folder:

1. **Scrape Service Set** (Shodan):
   Retrieves the top ``NUM_PRODUCTS`` services from the output of the provided query.

    **Output:** ``top_products.yaml``

2. **Scrape Service Version Set** (Shodan):
   For each service gathered from the previous step, scrape the top `NUM_VERSIONS_PER_PRODUCT` versions according to the desired query.

   **Output:** ``top_products_versions.yaml``

3. **Map Services to Use Cases**:
   Each service is mapped to one or more device types (e.g., IoT, Windows), representing the typical device type(s) for that service.

   **Mappings** are defined **manually** in: ``cyberbattle/data_generation/config/mappings.yaml``. Extend the mappings in case additional services than those contained are used.

4. **Scrape Vulnerabilities from NVD**:
   Fetches all known vulnerabilities for each service version determined above.

   **Output**: ``products_info.json``

5. **Classify Vulnerabilities**:
   Uses a classifier (configured via ``config.yaml``) to map each vulnerability to the corresponding MITRE ATT&CK Tactics it enables.

   **Output:** Adds a ``classes`` field to each vulnerability entry in ``products_info.json``.

6. **Extract NLP Embeddings**:
   Extracts vector representations for both services and vulnerabilities using one or more NLP extractors.

   **Output**: ``extracted_data_<NLP_MODEL>.json``

**Why are these steps necessary?**

- **Allocation Realism**: The number of devices (device count) per service version is recorded to estimate how frequently service versions appear and their importance in the query results. This information is then used to statistically align service assignments—services that appear more frequently in the query are assigned more often. Additionally, mapping services to device types ensures that services are only assigned to compatible device types as defined.
- **Semantic Representation**: NLP embeddings enable vulnerabilities and services to be incorporated into the graph as numerical features within node and edge feature vectors.
- **Vulnerability Outcome Approximation**: Vulnerability classes will determine how the agent can use the vulnerability.  If the agent selects the correct MITRE ATT&CK tactics that correspond to how a vulnerability can be exploited, its choice is considered valid.

Environment Database Integration
-----------------

Specify which scraped database to use for scenario generation by setting the ``nvd_data_path`` in the root ``config.yaml`` file:

.. code-block:: yaml

    nvd_data_path: nvd_folder_name

A default database is provided and can be downloaded using the ``setup.py`` script, but it can be replaced with a new one by following the steps above.

Environment Database Structure
-------------------------

The final environment database will be composed of the set of files ``extracted_data_<NLP_MODEL>.json``.
A simplified example of a service entry in the environment database is shown below:

.. code-block:: json

    {
        "product": "OpenSSH",
        "version": "7.4",
        "frequency": 1230,
        "tags": ["unix"],
        "description": "OpenSSH 7.4 ...",
        ...
        "feature_vector_BERT": [0.213, 4.12, ..., 0.12],
        ...
        "vulnerabilities": {
            "CVE-2023-51767": { .... },
            ....
        }
    }

- **count**: Number of internet nodes hosting this service version on the output query of the database (according to Shodan at the snapshot time).
- **description**: Brief textual description of the service.
- **feature_vector_MODEL**: NLP-generated embeddings representing the service description semantics.

Similarly, vulnerability entries include essential metadata:

.. code-block:: json

    {
        "cve_id": "CVE-2023-51767",
        "metrics": {
            ...
                "privilegesRequired": "NONE",
                "confidentialityImpact": "NONE",
            ...
        },
        "classes": [
            {
                "class": "credential access",
                "probability: 0.98
            },
        ],
        "description": "scp.c in the scp client allows remote SSH servers to bypass intended access restrictions.",
        ...
        "feature_vector_BERT": [0.743, 1.2342, ..., 0.433],
        ...
    }

- **metrics**: Vulnerability metrics and metadata from the NVD, including the Common Vulnerability Scoring System.
- **classes**: MITRE ATT&CK tactics associated, predicted via a multi-label classifier. See `MITRE ATT&CK Tactics Classification <mitre_classification.html>`_ for details.
- **description**: Free-text vulnerability summary.
- **feature_vector_...**: NLP embeddings capturing semantic content of the vulnerability.


Default Environment Database
------------------------

The **default database** includes the 50 most vulnerable services identified by Shodan. For each service, up to five of the most vulnerable versions were selected. Due to data availability, some services have fewer than five versions, resulting in a total of **172 service versions**.
This process uncovered **829 unique vulnerabilities** from the NVD, each affecting one or more service versions.
The services identified from the Shodan query (as of **18 November 2024**) and the simulated node types are:

**Services:**
nginx, Apache httpd, Squid http proxy, Microsoft IIS httpd, Exim smtpd, Jetty, MongoDB, Remote Desktop Protocol, Jenkins, Wildix Collaboration, Apache Tomcat, NET-DK, Metabase, Outlook Web App, Tengine, VMware ESXi, OpenSSH, OpenResty, MySQL, Control Web Panel, Apache Tomcat/Coyote JSP engine, Nextcloud, Elastic, lighttpd, GoAhead Embedded Web Server, Hikvision IP Camera, RabbitMQ, Boa Web Server, Grafana, micro_httpd, Boa HTTPd, DrayTek Vigor Series (2925, 2862, 2860, 2762, 2912, 2926, 2133, 2927, 2865), ZTE H268A, ZTE ZXHN H168N, Cisco Systems, ZTE F680, Bbox.

**Node Types:**
Windows Host, Unix Host, IoT node, Industrial Control System (ICS) node, Router.
