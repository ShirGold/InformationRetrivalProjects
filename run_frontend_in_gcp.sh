INSTANCE_NAME="instance-1"
REGION= "europe-west4"
ZONE= "europe-west4-c"
PROJECT_NAME="information-retrival-370211"
IP_NAME="shirg0001"
GOOGLE_ACCOUNT_NAME="shir.goldfarb" # without the @post.bgu.ac.il or @gmail.com part

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project $PROJECT_NAME
# gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create "shirg0001" --project="information-retrival-370211" --region="europe-west4"
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
INSTANCE_IP="shirg0001"

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 --allow tcp:8080 --source-ranges 0.0.0.0/0 --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create "instance-1" --zone="europe-west4-c" --machine-type=e2-micro --network-interface=address="shirg0001",network-tier=PREMIUM,subnet=default --metadata-from-file startup-script="C:\Users\shirg\Documents\Programming\Study\3rd\Information Retrival\Project\Project files\startup_script_gcp.sh" --scopes=https://www.googleapis.com/auth/cloud-platform --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output "instance-1" --zone "europe-west4-c"

# 4. Secure copy your app to the VM
gcloud compute scp "C:\Users\shirg\Documents\Programming\Study\3rd\Information Retrival\Project\Project files\search_frontend.py" "shir.goldfarb"@"instance-1":/home/"shir.goldfarb"
gcloud compute scp "C:\Users\shirg\Documents\Programming\Study\3rd\Information Retrival\Project\Project files\backend.py" "shir.goldfarb"@"instance-1":/home/"shir.goldfarb"
gcloud compute scp "C:\Users\shirg\Documents\Programming\Study\3rd\Information Retrival\Project\Project files\inverted_index_gcp.py" "shir.goldfarb"@"instance-1":/home/"shir.goldfarb"


# 5. SSH to your VM and start the app
gcloud compute ssh "shir.goldfarb"@"instance-1"
python3 search_frontend.py

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q "instance-1"
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses
gcloud compute addresses delete -q "shirg0001" --region "europe-west4"