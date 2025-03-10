# Batch compute environment (Fargate)
resource "aws_batch_compute_environment" "pipeline_fargate_env" {
  compute_environment_name = "${var.environment}-${var.system_name}-batch-compute-env"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = var.batch_service_role_arn

  compute_resources {
    type               = "FARGATE"
    max_vcpus          = var.max_vcpus
    subnets            = var.subnet_ids
    security_group_ids = [var.security_group_id]
  }

  tags = var.common_tags
}

resource "aws_batch_job_queue" "pipeline_job_queue" {
  name     = "${var.environment}-${var.system_name}-job-queue"
  state    = "ENABLED"
  priority = var.job_queue_priority

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.pipeline_fargate_env.arn
  }

  tags = var.common_tags
}


resource "aws_batch_job_definition" "pipeline_job_definition" {
  name = "${var.environment}-${var.system_name}-job-definition"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image = var.container_image_uri

    jobRoleArn       = var.batch_job_role_arn
    executionRoleArn = var.batch_execution_role_arn

    # Fargate requires explicit CPU and memory specification
    resourceRequirements = [
      {
        type  = "VCPU"
        value = tostring(var.job_vcpus)
      },
      {
        type  = "MEMORY"
        value = tostring(var.job_memory)
      }
    ]

    command = [
      ".venv/bin/python",
      "-m",
      "hsr_data_converter.rosbag2lerobot.main",
      "--secret_name",
      "Ref::SECRET_NAME",
      "--rosbags_bucket_name",
      "Ref::ROS_BAGS_BUCKET_NAME",
      "--lerobot_bucket_name",
      "Ref::LEROBOT_BUCKET_NAME",
      "--template_dir",
      "Ref::TEMPLATE_DIR",
      "--fps",
      "Ref::FPS",
      "--use_aws",
      "true"
    ]

    parameters = {
      "TEMPLATE_DIR"         = var.template_dir,
      "SECRET_NAME"          = var.secret_name,
      "LEROBOT_BUCKET_NAME"  = var.lerobot_bucket_name,
      "ROS_BAGS_BUCKET_NAME" = var.ros_bags_bucket_name,
      "FPS"                  = var.fps,
    }

    environment = []
    secrets     = [],


    networkConfiguration = {
      assignPublicIp = "ENABLED"
    },

    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    },

    ephemeralStorage = {
      sizeInGiB = 100
    },

    runtimePlatform = {
      operatingSystemFamily = "LINUX",
      cpuArchitecture       = "X86_64"
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch_logs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  })

  tags = var.common_tags
}


# CloudWatch log group for Batch jobs
resource "aws_cloudwatch_log_group" "batch_logs" {
  name              = "/aws/batch/job/${var.environment}-${var.system_name}-batch-logs"
  retention_in_days = var.log_retention_days

  tags = var.common_tags
}
