output "secret_arn" {
  description = "ARN of the secret"
  value       = data.aws_secretsmanager_secret.wasabi_credentials.arn
}

output "secret_name" {
  description = "Name of the secret"
  value       = data.aws_secretsmanager_secret.wasabi_credentials.name
}
