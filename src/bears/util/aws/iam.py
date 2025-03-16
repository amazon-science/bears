from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError
from pydantic import constr

from bears.util.language import Alias, String, as_list, filter_kwargs, get_default, safe_validate_arguments
from bears.util.logging import Log


class IAMUtil:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    @classmethod
    @safe_validate_arguments
    def assume_role(
        cls,
        role_arn: Union[str, List[str]],
        *,
        session_name: Optional[constr(min_length=1)] = None,
        credentials: Optional[Dict[str, str]] = None,
        intermediate: bool = False,
        try_set_max_duration: bool = False,
        default_max_duration: int = 3600,
    ) -> Union[Dict, List[Dict]]:
        """
        Assume one or more IAM roles (chain them) using boto3's STS client.
        For each role, if try_set_max_duration is True, this function tries to query
        the role's MaxSessionDuration via IAM.get_role and then passes that value as DurationSeconds.
        If the call fails (e.g. due to insufficient permissions), it falls back to not passing DurationSeconds.

        :param role_arn: ARN (or list of ARNs) of the role(s) to assume.
        :param session_name: A unique session name for the assume-role call.
        :param credentials: (Optional) A dictionary of credentials to use before assuming the first role.
                            If not provided, boto3's default credential chain is used.
        :param intermediate: Whether to return all intermediate credentials and other metadata.
        :param try_set_max_duration: If True, attempt to set DurationSeconds to the role's maximum.
        :param default_max_duration: The default DurationSeconds to use if try_set_max_duration is False,
            or if it fails. Default value is 3600 i.e. 1 hour.
        :return: A dict mapping each role ARN to a dict with keys "credentials" and "max_duration".
        """
        session_name: str = get_default(
            session_name, String.convert_case(String.random_name(), target_case="pascal")
        )

        role_arn: List[str] = as_list(role_arn)
        if not role_arn:
            raise ValueError("No role ARN provided.")

        all_credentials: List[Dict[str, Any]] = []
        for role in role_arn:
            # Create STS and IAM clients using provided credentials or default chain.
            if credentials is not None:
                sts_client = boto3.client(
                    "sts",
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"],
                )
                iam_client = boto3.client(
                    "iam",
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"],
                )
            else:
                sts_client = boto3.client("sts")
                iam_client = boto3.client("iam")

            max_duration: Optional[int] = None
            if try_set_max_duration:
                # Extract role name from ARN (assumes ARN format contains '/RoleName')
                role_name = role.split("/")[-1]
                try:
                    role_details: Dict[str, Any] = iam_client.get_role(RoleName=role_name)
                    max_duration = role_details["Role"]["MaxSessionDuration"]
                except ClientError as e:
                    # If we cannot retrieve the role details, log and fallback.
                    Log.warning(
                        f"Unable to retrieve MaxSessionDuration for {role}, "
                        f"falling back to default of {default_max_duration}. "
                        f"Error: {e}"
                    )
                    max_duration = None
            if max_duration is None:
                max_duration: int = default_max_duration

            # Build parameters for assume_role call.
            assume_params = {
                "RoleArn": role,
                "RoleSessionName": session_name,
                "DurationSeconds": max_duration,
            }
            response: Dict[str, Any] = sts_client.assume_role(**assume_params)
            credentials = response["Credentials"]
            all_credentials.append(
                {
                    **assume_params,
                    "Credentials": credentials,
                }
            )

        if intermediate:
            return all_credentials
        ## Return only the final role's credentials wrapped in our nested dict format.
        return all_credentials[-1]

    @classmethod
    @safe_validate_arguments
    def create_session(
        cls,
        *,
        role_arn: Optional[Union[str, List[str]]] = None,
        credentials: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> boto3.Session:
        """
        Create a boto3 session with the given credentials and optional role ARN.
        If a role ARN is provided, this function will call IAMUtil.assume_role to assume the role.
        If a region is provided, it will be set in the session configuration.

        :param role_arn: ARN (or list of ARNs) of the role(s) to assume.
        :param credentials: (Optional) A dictionary of credentials to use before assuming the first role.
                            If not provided, boto3's default credential chain is used.
        :param kwargs: Additional arguments to pass to boto3.Session
        :return: A boto3.Session object.
        """
        Alias.set_region_name(kwargs)

        ## Set config params:
        session_config: Dict = {}

        region_name: str = kwargs.get("region_name", None)
        if region_name is not None:
            session_config["region_name"] = region_name

        if role_arn is not None:
            credentials: Dict[str, str] = cls.assume_role(
                role_arn,
                credentials=credentials,
                **filter_kwargs(cls.assume_role, **kwargs),
            )["Credentials"]
            session_config["aws_access_key_id"] = credentials["AccessKeyId"]
            session_config["aws_secret_access_key"] = credentials["SecretAccessKey"]
            session_config["aws_session_token"] = credentials["SessionToken"]

        return boto3.Session(**session_config)
