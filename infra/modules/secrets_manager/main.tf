# Secrets Manager secret for Wasabi credentials
data "aws_secretsmanager_secret" "wasabi_credentials" {
  name = var.secret_name
}
